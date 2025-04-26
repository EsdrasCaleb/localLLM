package com.hf.sfm.util;

import java.util.ArrayList;
import java.util.HashMap;
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
import java.util.Iterator;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

class Loader_collectToMap_16_0_Test {

    private Loader loader;

    @BeforeEach
    void setUp() {
        loader = new Loader();
        // Mock total count
        loader.setTotalCount(2);
    }

    @Test
    void testCollectToMapWithComboFlag() {
        // Prepare mock data
        List<Object[]> mockRs = new ArrayList<>();
        mockRs.add(new Object[] { "1", "Item 1" });
        mockRs.add(new Object[] { "2", "Item 2" });
        loader.setRs(mockRs);
        // Call the method under test
        loader.collectToMap("combo");
        // Verify the results
        ListRange range = loader.getRange();
        assertNotNull(range);
        assertEquals(2, range.getTotalSize());
        List<Object> data = range.getData();
        assertEquals(2, data.size());
        HashMap<String, Object> firstRow = (HashMap<String, Object>) data.get(0);
        assertEquals("1", firstRow.get("value"));
        assertEquals("Item 1", firstRow.get("text"));
        HashMap<String, Object> secondRow = (HashMap<String, Object>) data.get(1);
        assertEquals("2", secondRow.get("value"));
        assertEquals("Item 2", secondRow.get("text"));
    }

    @Test
    void testCollectToMapWithoutComboFlag() {
        // Prepare mock data
        List<Object[]> mockRs = new ArrayList<>();
        mockRs.add(new Object[] { "1", "Item 1", "Extra Data 1" });
        mockRs.add(new Object[] { "2", "Item 2", "Extra Data 2" });
        String[] colNames = { "id", "name", "extra" };
        loader.setColNames(colNames);
        loader.setRs(mockRs);
        // Call the method under test
        loader.collectToMap("");
        // Verify the results
        ListRange range = loader.getRange();
        assertNotNull(range);
        assertEquals(2, range.getTotalSize());
        List<Object> data = range.getData();
        assertEquals(2, data.size());
        HashMap<String, Object> firstRow = (HashMap<String, Object>) data.get(0);
        assertEquals("1", firstRow.get("id"));
        assertEquals("Item 1", firstRow.get("name"));
        assertEquals("Extra Data 1", firstRow.get("extra"));
        HashMap<String, Object> secondRow = (HashMap<String, Object>) data.get(1);
        assertEquals("2", secondRow.get("id"));
        assertEquals("Item 2", secondRow.get("name"));
        assertEquals("Extra Data 2", secondRow.get("extra"));
    }

    @Test
    void testCollectToMapWithEmptyResultSet() {
        // Prepare empty result set
        loader.setRs(new ArrayList<>());
        // Call the method under test
        loader.collectToMap("combo");
        // Verify the results
        ListRange range = loader.getRange();
        assertNotNull(range);
        assertEquals(0, range.getTotalSize());
        assertTrue(range.getData().isEmpty());
    }
}
