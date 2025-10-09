const getAxios = require("../config/axios");
const models = require("../models/models"); // Путь к модели

const baseUrl =
    "http://cris.icc.ru/dataset/list?f=100&count_rows=true&iDisplayStart=0&iDisplayLength=1";

const createBaseUrl = (displayStart, displayLength) => {
    return `http://cris.icc.ru/dataset/list?f=100&count_rows=true&iDisplayStart=${displayStart}&iDisplayLength=${displayLength}`;
};

const getMinMaxId = async () => {
    try {
        const maxId = await models.Dataset.max("id");
        const minId = await models.Dataset.min("id");

        return { minId, maxId };
    } catch (error) {
        console.error("Error fetching or saving data:", error);
    }
};

const updateDatasets = async (req, res) => {
    const requestData = {
        sort: [{ fieldname: "id", dir: false }],
    };

    const axios = getAxios();

    const response = await axios
        .post(`${baseUrl}`, requestData)
        .catch((err) => console.log(err));

    const datasetWithMaxIdInRemoteServer = response.data;
    const minMaxIdInDataBase = await getMinMaxId();

    const totalRecords = +datasetWithMaxIdInRemoteServer.iTotalDisplayRecords;

    const datasetData = await models.Dataset.findAll();
    // console.log("максимальный id RS", maxIdInRemoteServer);
    console.log("minmax id DB", minMaxIdInDataBase);
    console.log("calls on RS", totalRecords);
    // console.log("длинна массива базы данных", callWithMaxIdInRemoteServer);
    console.log("длинна массива базы данных", datasetData.length);

    const datasetDataIds = datasetData.map((call) => call.id);

    const displayLength = 500;
    let iDisplayStart = 0;
    let counter = 1;

    try {
        while (iDisplayStart < totalRecords ) {
            console.log("datasetData update counter", counter);
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
                const [dataset, created] = await models.Dataset.findOrCreate({
                    where: { id: item.id },
                    defaults: {
                        id: item.id,
                        guid: item.guid
                    },
                });
                if (created) {
                    console.log("dataset already exist");
                }
            }

            iDisplayStart += displayLength;

            if (data.length < displayLength) {
                console.log("datasets закончились");
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

module.exports = {
    updateDatasets
};
