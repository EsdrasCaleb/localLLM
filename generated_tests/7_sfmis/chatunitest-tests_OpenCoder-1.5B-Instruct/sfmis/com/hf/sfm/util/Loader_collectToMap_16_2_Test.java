package com.hf.sfm.util;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

@RunWith(MockitoJUnitRunner.class)
public class Loader_collectToMap_16_2_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private Loader loaderTest;

    @Test
    public void collectToMapTest() {
        // Arrange
        List<Map<String, Object>> expected = new ArrayList<>();
        Map<String, Object> map = new HashMap<>();
        map.put("value", "1");
        map.put("text", "One");
        expected.add(map);
        when(loader.getRs()).thenReturn(expected);
        when(loader.getTotalCount()).thenReturn(1);
        // Act
        loaderTest.collectToMap("combo");
        // Assert
        Mockito.verify(loader).getRs();
        Mockito.verify(loader).getTotalCount();
        // Add assertions to check the expected result against the actual result
    }
}
