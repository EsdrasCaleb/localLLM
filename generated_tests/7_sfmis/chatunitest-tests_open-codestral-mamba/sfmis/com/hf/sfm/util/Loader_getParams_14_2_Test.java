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

public class Loader_getParams_14_2_Test {

    @Mock
    private Query queryMock;

    @InjectMocks
    private Loader loader;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetParams() {
        String[][] params = { { "1", "2" }, { "String", "String" } };
        when(queryMock.setParameter(0, "1")).thenReturn(queryMock);
        when(queryMock.setParameter(1, "2")).thenReturn(queryMock);
        Query result = loader.getParams(queryMock, params);
        verify(queryMock, times(2)).setParameter(anyInt(), anyString());
        // Additional assertions can be added here to validate the result
    }
}
