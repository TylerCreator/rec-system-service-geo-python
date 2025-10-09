const axios = require('axios')
const https = require('https')
let instance

module.exports = function getAxios()
{
    if (!instance)
    {
        //create axios instance
        instance = axios.create({
            timeout: 90000, //optional
            headers: {'Content-Type':'json'}
        })
    }

    return instance;
}