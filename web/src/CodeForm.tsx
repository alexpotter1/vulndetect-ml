import React, { useState } from 'react';
import { useFormik } from 'formik';
import { makeStyles, Button } from '@material-ui/core';
import { ICodeVulnerableProps } from './Detector';
import APIClient from './api/APIClient';

const useStyles = makeStyles((theme) => ({
    boxFullWidth: {
        '-webkit-box-sizing': 'border-box',
        '-moz-box-sizing': 'border-box',
        'box-sizing': 'border-box',
        'width': '100%',
    }
}));

const CodeForm = (props: ICodeVulnerableProps) => {
    const classes = useStyles();
    const [codeFile, setCodeFile] = useState<File | undefined>(undefined)
    const formik = useFormik({
        initialValues: {
            code: '',
        },
        onSubmit: values => {
            let data = new FormData();
            if (values.code.length > 0) {
                data.append('file', values.code)
            } else if (codeFile !== undefined) {
                data.append('file', codeFile)
            } else {
                console.log('Error: no code entered/uploaded!')
                return;
            }

            APIClient.postCodeSample(data).then((res) => {
                console.log(res.status)
                console.log(res.message)
            })
        },
    });

    return (
        <form onSubmit={formik.handleSubmit}>
            <label htmlFor="code">Source Code</label>
            <div>
                <textarea
                    className={classes.boxFullWidth}
                    id="code"
                    name="code"
                    rows={20}
                    cols={100}
                    onChange={formik.handleChange}
                    value={formik.values.code}></textarea>
            </div>
            <input className={classes.boxFullWidth} id="file" name="file" type="file" onChange={(event) => {
                if (event.currentTarget.files !== null && event.currentTarget.files.length === 1) {
                    setCodeFile(event.currentTarget.files[0])
                }
            }}/>
            <Button color="secondary" variant="contained" type="submit">Submit</Button>
        </form>
    )
}

export default CodeForm;