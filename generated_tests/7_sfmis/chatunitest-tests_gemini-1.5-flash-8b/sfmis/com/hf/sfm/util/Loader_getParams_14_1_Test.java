package com.hf.sfm.util;

import org.hibernate.Query;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Arrays;
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
import org.hibernate.HibernateException;
import org.hibernate.Session;

class Loader_getParams_14_1_Test {

    @Test
    void testGetParams_validInput() throws ParseException {
        // Mock Query object
        Query mockQuery = Mockito.mock(Query.class);
        // Input parameters
        String[][] params = { { "1", "2", "3", "4", "2023-10-26" }, { "String", "Integer", "Long", "Double", "Date" } };
        // Expected parameters
        Mockito.when(mockQuery.setParameter(0, "1")).thenReturn(mockQuery);
        Mockito.when(mockQuery.setParameter(1, 2)).thenReturn(mockQuery);
        Mockito.when(mockQuery.setParameter(2, 3L)).thenReturn(mockQuery);
        Mockito.when(mockQuery.setParameter(3, 4.0)).thenReturn(mockQuery);
        Mockito.when(mockQuery.setParameter(4, new SimpleDateFormat("yyyy-MM-dd").parse("2023-10-26"))).thenReturn(mockQuery);
        // Call the method under test
        Loader loader = new Loader();
        Query resultQuery = loader.getParams(mockQuery, params);
        // Verify that the mockQuery was called with the expected parameters
        Mockito.verify(mockQuery).setParameter(0, "1");
        Mockito.verify(mockQuery).setParameter(1, 2);
        Mockito.verify(mockQuery).setParameter(2, 3L);
        Mockito.verify(mockQuery).setParameter(3, 4.0);
        Mockito.verify(mockQuery).setParameter(4, new SimpleDateFormat("yyyy-MM-dd").parse("2023-10-26"));
        // Assert the returned query is the same as the mocked query
        assertSame(mockQuery, resultQuery);
    }

    @Test
    void testGetParams_nullParams() {
        // Mock Query object
        Query mockQuery = Mockito.mock(Query.class);
        Loader loader = new Loader();
        Query resultQuery = loader.getParams(mockQuery, null);
        assertSame(mockQuery, resultQuery);
        // No calls to setParameter should be made
        Mockito.verifyNoMoreInteractions(mockQuery);
    }

    @Test
    void testGetParams_emptyParams() {
        // Mock Query object
        Query mockQuery = Mockito.mock(Query.class);
        Loader loader = new Loader();
        String[][] params = { {}, {} };
        Query resultQuery = loader.getParams(mockQuery, params);
        assertSame(mockQuery, resultQuery);
        Mockito.verifyNoMoreInteractions(mockQuery);
    }
}
