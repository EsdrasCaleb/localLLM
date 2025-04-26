package com.hf.sfm.util;

import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
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

    @Test
    @ExtendWith(MockitoExtension.class)
    void runTest() {
        MockitoAnnotations.openMocks(this);
        Loader loader = Mockito.mock(Loader.class);
        BasePara basePara = Mockito.mock(BasePara.class);
        // Prepare mock data
        Mockito.when(basePara.getSqlpath()).thenReturn("/path/to/sql");
        Mockito.when(basePara.getStart()).thenReturn(0);
        Mockito.when(basePara.getLimit()).thenReturn(10);
        Mockito.when(basePara.getSort()).thenReturn("name");
        Mockito.when(basePara.getDir()).thenReturn("ASC");
        Mockito.when(basePara.isPaging()).thenReturn(true);
        Mockito.when(basePara.getQuerySql()).thenReturn("select * from users");
        Mockito.when(basePara.getQueryValue()).thenReturn("user1");
        // Execute the method under test
        loader.run(basePara);
        // Verify the results
        // Assert the expected values here
    }
}
