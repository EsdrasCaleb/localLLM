package com.hf.sfm.util;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

public class Loader_parseXML_9_1_Test {

    @InjectMocks
    private Loader loader;

    @Mock
    private SAXReader saxReader;

    @Mock
    private Document document;

    @Mock
    private Element rootElement;

    @Mock
    private Element mainSqlElement;

    @Mock
    private Element querySqlElement;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        loader = new Loader();
        Field filepathField = Loader.class.getDeclaredField("filepath");
        filepathField.setAccessible(true);
        filepathField.set(loader, "test//testElement");
    }

    @Test
    public void testParseXML() throws Exception {
        when(saxReader.read(any(File.class))).thenReturn(document);
        when(document.getRootElement()).thenReturn(rootElement);
        when(rootElement.elementIterator()).thenReturn(mock(Iterator.class));
        when(rootElement.elementIterator().hasNext()).thenReturn(true, false);
        when(rootElement.elementIterator().next()).thenReturn(mainSqlElement, querySqlElement);
        when(mainSqlElement.getName()).thenReturn("testElement");
        when(mainSqlElement.elementText("main_sql")).thenReturn("SELECT * FROM test");
        when(mainSqlElement.elementText("query_sql")).thenReturn("WHERE test = ?");
        loader.parseXML();
        Field sqlField = Loader.class.getDeclaredField("sql");
        sqlField.setAccessible(true);
        String sql = (String) sqlField.get(loader);
        assertEquals("SELECT * FROM test", sql);
        Field querySqlField = Loader.class.getDeclaredField("querySql");
        querySqlField.setAccessible(true);
        String querySql = (String) querySqlField.get(loader);
        assertEquals("WHERE test = ?", querySql);
    }
}
