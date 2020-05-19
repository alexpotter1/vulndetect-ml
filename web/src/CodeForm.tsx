import React, { useState, useEffect } from 'react';
import { useFormik } from 'formik';
import { makeStyles, Button } from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
    boxFullWidth: {
        '-webkit-box-sizing': 'border-box',
        '-moz-box-sizing': 'border-box',
        'box-sizing': 'border-box',
        'width': '100%',
    }
}));

const CodeForm = () => {
    const classes = useStyles();
    const [codeFile, setCodeFile] = useState<File | null>(null)
    const formik = useFormik({
        initialValues: {
            code: '',
        },
        onSubmit: values => {
            alert(JSON.stringify(values, null, 1))
        },
    });

    useEffect(() => {
        if (codeFile !== null) {
            let data = new FormData();
            data.append('file', codeFile)
        }
    }, [codeFile]);

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