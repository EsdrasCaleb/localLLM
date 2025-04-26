package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

public class Loader_run_8_2_Test {

    @Mock
    private Loader loader;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @AfterEach
    public void tearDown() {
    }

    @Test
    public void testRun() {
        BasePara basePara = Mockito.mock(BasePara.class);
        Mockito.when(basePara.getSqlpath()).thenReturn("path");
        Mockito.when(basePara.isPaging()).thenReturn(true);
        Mockito.when(basePara.getStart()).thenReturn(1);
        Mockito.when(basePara.getLimit()).thenReturn(10);
        Mockito.when(basePara.getSort()).thenReturn("sort");
        Mockito.when(basePara.getDir()).thenReturn("asc");
        Mockito.when(basePara.single2plannar()).thenReturn(null);
        Mockito.when(basePara.getQuerySql()).thenReturn("select * from table");
        Mockito.when(basePara.getQueryValue()).thenReturn("value");
        loader.run(basePara);
        // Add your assertions here
    }
}
