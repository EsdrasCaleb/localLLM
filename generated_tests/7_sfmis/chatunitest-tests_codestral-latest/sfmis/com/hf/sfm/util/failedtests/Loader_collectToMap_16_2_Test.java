package com.hf.sfm.util;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

class Loader_collectToMap_16_2_Test {

    private Loader loader;

    private List<Object[]> rs;

    private String[] colNames;

    private int totalCount;

    @BeforeEach
    void setUp() throws Exception {
        loader = new Loader();
        rs = new ArrayList<>();
        colNames = new String[] { "col1", "col2" };
        totalCount = 2;
        // Mocking the necessary fields using reflection
        Field rsField = Loader.class.getDeclaredField("rs");
        rsField.setAccessible(true);
        rsField.set(loader, rs);
        Field colNamesField = Loader.class.getDeclaredField("colNames");
        colNamesField.setAccessible(true);
        colNamesField.set(loader, colNames);
        Field totalCountField = Loader.class.getDeclaredField("totalCount");
        totalCountField.setAccessible(true);
        totalCountField.set(loader, totalCount);
    }

    @Test
    void testCollectToMap() throws Exception {
        // Prepare test data
        rs.add(new Object[] { "value1", "text1" });
        rs.add(new Object[] { "value2", "text2" });
        // Invoke the method
        loader.collectToMap("combo");
        // Verify the results
        ListRange range = loader.getRange();
        assertNotNull(range);
        assertEquals(totalCount, range.getTotalSize());
        List<Object> data = range.getData();
        assertNotNull(data);
        assertEquals(rs.size(), data.size());
        for (int i = 0; i < data.size(); i++) {
            HashMap<String, Object> map = (HashMap<String, Object>) data.get(i);
            assertEquals("value" + (i + 1), map.get("value"));
            assertEquals("text" + (i + 1), map.get("text"));
        }
    }

    @Test
    void testCollectToMapEmptyRs() throws Exception {
        // Prepare test data
        rs.clear();
        // Invoke the method
        loader.collectToMap("combo");
        // Verify the results
        ListRange range = loader.getRange();
        assertNotNull(range);
        assertEquals(totalCount, range.getTotalSize());
        List<Object> data = range.getData();
        assertNotNull(data);
        assertEquals(0, data.size());
    }

    @Test
    void testCollectToMapSingleColumn() throws Exception {
        // Prepare test data
        colNames = new String[] { "col1" };
        Field colNamesField = Loader.class.getDeclaredField("colNames");
        colNamesField.setAccessible(true);
        colNamesField.set(loader, colNames);
        rs.add(new Object[] { "value1" });
        rs.add(new Object[] { "value2" });
        // Invoke the method
        loader.collectToMap("combo");
        // Verify the results
        ListRange range = loader.getRange();
        assertNotNull(range);
        assertEquals(totalCount, range.getTotalSize());
        List<Object> data = range.getData();
        assertNotNull(data);
        assertEquals(rs.size(), data.size());
        for (int i = 0; i < data.size(); i++) {
            HashMap<String, Object> map = (HashMap<String, Object>) data.get(i);
            assertEquals("value" + (i + 1), map.get("col1"));
        }
    }
}
