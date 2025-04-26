package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

@ExtendWith(MockitoExtension.class)
public class Loader_collectToMap_16_2_Test {

    @Mock
    private Loader loader;

    @Test
    public void testCollectToMap() {
        when(loader.getColNames()).thenReturn(new String[] { "id", "name" });
        when(loader.getRs()).thenReturn(Arrays.asList(new Object[] { "1", "John" }, new Object[] { "2", "Jane" }));
        // Test the method
        loader.collectToMap("normal");
        ListRange range = loader.getRange();
        assertNotNull(range, "Range should not be null");
        assertEquals(2, range.getData().size(), "Data size should be 2");
        assertEquals(10, range.getTotalSize(), "Total size should be 10");
    }
}
