package com.hf.sfm.util;

import org.hibernate.Query;
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
import org.hibernate.Session;

public class Loader_getParams_14_0_Test {

    private Loader loader;

    private Query mockQuery;

    @BeforeEach
    public void setUp() {
        loader = new Loader();
        mockQuery = mock(Query.class);
    }

    @Test
    public void testGetParamsWithStringParameter() {
        String[][] params = { { "test" }, { "String" } };
        Query result = loader.getParams(mockQuery, params);
        verify(mockQuery).setParameter(0, "test");
        assertEquals(mockQuery, result);
    }

    @Test
    public void testGetParamsWithLongParameter() {
        String[][] params = { { "123" }, { "Long" } };
        Query result = loader.getParams(mockQuery, params);
        verify(mockQuery).setParameter(0, 123L);
        assertEquals(mockQuery, result);
    }

    @Test
    public void testGetParamsWithIntegerParameter() {
        String[][] params = { { "456" }, { "Integer" } };
        Query result = loader.getParams(mockQuery, params);
        verify(mockQuery).setParameter(0, 456);
        assertEquals(mockQuery, result);
    }

    @Test
    public void testGetParamsWithDoubleParameter() {
        String[][] params = { { "78.9" }, { "Double" } };
        Query result = loader.getParams(mockQuery, params);
        verify(mockQuery).setParameter(0, 78.9);
        assertEquals(mockQuery, result);
    }

    @Test
    public void testGetParamsWithDateParameter() throws Exception {
        String[][] params = { { "2023-10-01" }, { "Date" } };
        Query result = loader.getParams(mockQuery, params);
        verify(mockQuery).setParameter(0, new SimpleDateFormat("yyyy-MM-dd").parse("2023-10-01"));
        assertEquals(mockQuery, result);
    }

    @Test
    public void testGetParamsWithNullParams() {
        Query result = loader.getParams(mockQuery, null);
        assertEquals(mockQuery, result);
        verify(mockQuery, never()).setParameter(anyInt(), any());
    }

    @Test
    public void testGetParamsWithEmptyParams() {
        String[][] params = { {}, {} };
        Query result = loader.getParams(mockQuery, params);
        assertEquals(mockQuery, result);
        verify(mockQuery, never()).setParameter(anyInt(), any());
    }
}
