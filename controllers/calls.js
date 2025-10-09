const getAxios = require("../config/axios");
const models = require("../models/models"); // Путь к модели
// const axios = require("axios")
const { Parser } = require("@json2csv/plainjs");
const fs = require("fs");

const url =
    "http://cris.icc.ru/dataset/list?f=186&count_rows=true&unique=undefined&count_rows=1&iDisplayStart=0&iDisplayLength=100";
const baseUrl2 = "http://cris.icc.ru/dataset/list?f=186&count_rows=true";

const baseUrl =
    "http://cris.icc.ru/dataset/list?f=186&count_rows=true&iDisplayStart=0&iDisplayLength=1";

const createBaseUrl = (displayStart, displayLength) => {
    return `http://cris.icc.ru/dataset/list?f=186&count_rows=true&iDisplayStart=${displayStart}&iDisplayLength=${displayLength}`;
};

const deleteCallsByIds = async (idsToDelete) => {
    try {
        // Удалите записи с указанными id
        const deletedRowsCount = await models.Call.destroy({
            where: {
                id: idsToDelete,
            },
        });

        console.log(`Deleted ${deletedRowsCount} rows from the 'Call' table.`);
        return deletedRowsCount; // Вернуть количество удаленных строк
    } catch (error) {
        console.error("Error deleting data:", error);
        throw error; // Пробрасываем ошибку выше для дальнейшей обработки
    }
};

const getMinMaxId = async () => {
    try {
        const maxId = await models.Call.max("id");
        const minId = await models.Call.min("id");
        console.log("Max ID:", maxId);
        console.log("Min ID:", minId);
        return { minId, maxId };
    } catch (error) {
        console.error("Error fetching or saving data:", error);
    }
};

const updateCalls = async (req, res) => {
    const requestData = {
        sort: [{ fieldname: "id", dir: false }],
    };

    const axios = getAxios();

    const response = await axios
        .post(`${baseUrl}`, requestData)
        .catch((err) => console.log(err));

    const callWithMaxIdInRemoteServer = response.data;
    const minMaxIdInDataBase = await getMinMaxId();

    const maxIdInRemoteServer = callWithMaxIdInRemoteServer.aaData[0].id;
    const totalRecords = +callWithMaxIdInRemoteServer.iTotalDisplayRecords;

    const callData = await models.Call.findAll();
    // console.log("максимальный id RS", maxIdInRemoteServer);
    console.log("minmax id DB", minMaxIdInDataBase);
    console.log("calls on RS", totalRecords);
    // console.log("длинна массива базы данных", callWithMaxIdInRemoteServer);
    console.log("длинна массива базы данных", callData.length);

    const callDataIds = callData.map((call) => call.id);

    const displayLength = 500;
    let iDisplayStart = 0;
    let counter = 1;

    try {
        while (iDisplayStart < totalRecords ) {
            console.log("calls update counter", counter);
            console.log(iDisplayStart, " ----------------------------------");
            const requestUrl = createBaseUrl(iDisplayStart, displayLength);

            console.log("requestUrl", requestUrl);
            const response = await axios
                .post(requestUrl, requestData)
                .catch((e) => {
                    console.log("catch", e);
                });
            if (!response) continue;
            const data = response.data.aaData;

            if (data.length === 0) {
                console.log("no data");
                break;
            }

            for (const item of data) {
                const [call, created] = await models.Call.findOrCreate({
                    where: { id: item.id },
                    defaults: {
                        id: item.id,
                        classname: item.classname,
                        console_output: item.console_output,
                        created_by: item.created_by,
                        created_on: item.created_on,
                        edited_by: item.edited_by,
                        edited_on: item.edited_on,
                        end_time: item.end_time,
                        error_output: item.error_output,
                        input: item.input,
                        input_data: item.input_data,
                        input_params: item.input_params,
                        is_deleted: item.is_deleted,
                        mid: item.mid,
                        os_pid: item.os_pid,
                        owner: item.owner,
                        result: item.result,
                        status: item.status,
                        start_time: item.start_time,
                    },
                });
                if (created) {
                    console.log("service call already exist");
                }
            }

            iDisplayStart += displayLength;

            if (data.length < displayLength) {
                console.log("calls закончились");
                break;
            }
            counter += 1;
        }

        console.log("Data synchronization completed.");
    } catch (error) {
        console.log(iDisplayStart, " ----------------------------------");
        console.error("Error during data synchronization:", error);
        // res.status(400).send(error);
        throw error;
    }
};

const incrCalls = async (req, res) => {
    try {
        const response = await axios.get(url);
        const data = response.data.aaData;

        // Преобразование данных и сохранение в базе данных
        await models.Call.bulkCreate(
            data.map((item) => ({
                classname: item.classname,
                console_output: item.console_output,
                created_by: item.created_by,
                created_on: item.created_on,
                edited_by: item.edited_by,
                edited_on: item.edited_on,
                end_time: item.end_time,
                error_output: item.error_output,
                id: item.id,
                input: item.input,
                input_data: item.input_data,
                input_params: item.input_params,
                is_deleted: item.is_deleted,
                mid: item.mid,
                os_pid: item.os_pid,
                owner: item.owner,
                result: item.result,
                start_time: item.start_time,
                status: item.status,
            }))
        );
        res.send({ data: data });
        console.log("Data saved to the database.");
    } catch (error) {
        console.error("Error fetching or saving data:", error);
    }
};

const getCalls = async (req, res) => {
    console.log("getted");
    try {
        const callData = await models.Call.findAll({
            order: [["id", "DESC"]],
        });

        res.send(callData);
        console.log("Call data from the database:", callData.length);
    } catch (error) {
        console.error("Error fetching data from the database:", error);
        throw error; // Пробрасываем ошибку выше для дальнейшей обработки
    }
};

const dumpCsv = async (req, res) => {
    console.log("dump-csv");
    try {
        await models.Call.findAll().then((objs) => {
            let calls = [];
            console.log("allfound", objs.length);
            objs.forEach((obj) => {
                const { id, mid, owner, start_time } = obj;
                calls.push({ id, mid, owner, start_time });
            });

            const json2csvParser = new Parser({ header: true, delimiter: ";" });
            const csv = json2csvParser.parse(calls);
            fs.writeFile("calls.csv", csv, function (error) {
                if (error) throw error;
                console.log("Write to calls.csv successfully!");
            });
        });
        res.send(200);
        console.log("csv created");
    } catch (e) {
        console.log('Something went wrong while generating csv!')
    }
   
    
};

// updateCalls();

module.exports = {
    getCalls,
    incrCalls,
    updateCalls,
    dumpCsv,
};
