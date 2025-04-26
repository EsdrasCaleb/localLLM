package com.hf.sfm.util;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.Session;

class Loader_getParams_14_0_Test {

    private Loader loader;

    private Query query;

    @BeforeEach
    void setUp() {
        loader = new Loader();
        query = mock(Query.class);
    }

    @Test
    void testGetParamsWithStringType() {
        String[][] params = { { "test" }, { "String" } };
        Query resultQuery = loader.getParams(query, params);
        verify(query).setParameter(0, "test");
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithLongType() {
        String[][] params = { { "12345" }, { "Long" } };
        Query resultQuery = loader.getParams(query, params);
        verify(query).setParameter(0, 12345L);
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithIntegerType() {
        String[][] params = { { "123" }, { "Integer" } };
        Query resultQuery = loader.getParams(query, params);
        verify(query).setParameter(0, 123);
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithDoubleType() {
        String[][] params = { { "123.45" }, { "Double" } };
        Query resultQuery = loader.getParams(query, params);
        verify(query).setParameter(0, 123.45);
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithDateType() throws ParseException {
        String[][] params = { { "2023-10-01" }, { "Date" } };
        Query resultQuery = loader.getParams(query, params);
        Date expectedDate = new SimpleDateFormat("yyyy-MM-dd").parse("2023-10-01");
        verify(query).setParameter(0, expectedDate);
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithNullParams() {
        Query resultQuery = loader.getParams(query, null);
        verifyNoInteractions(query);
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithEmptyParams() {
        String[][] params = { {}, {} };
        Query resultQuery = loader.getParams(query, params);
        verifyNoInteractions(query);
        assertEquals(query, resultQuery);
    }

    @Test
    void testGetParamsWithInvalidDateFormat() throws ParseException {
        String[][] params = { { "invalid-date" }, { "Date" } };
        Query resultQuery = loader.getParams(query, params);
        verify(query).setParameter(0, null);
        assertEquals(query, resultQuery);
    }
}
