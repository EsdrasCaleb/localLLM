package com.hf.sfm.util;

import com.hf.sfm.util.Loader;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
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
class Loader_loadDataWithSql_12_4_Test {

    @Mock
    Loader loader;

    @Test
    void testLoadDataWithSql() {
        // Arrange
        List<Object[]> expectedData = new ArrayList<>();
        expectedData.add(new Object[] { 1, "John", "Doe" });
        expectedData.add(new Object[] { 2, "Jane", "Doe" });
        List<Object[]> mockedData = new ArrayList<>();
        mockedData.add(new Object[] { 1, "John", "Doe" });
        mockedData.add(new Object[] { 2, "Jane", "Doe" });
        when(loader.loadDataWithSql()).thenReturn(mockedData);
        // Act
        List<Object[]> actualData = loader.loadDataWithSql();
        // Assert
        assertEquals(expectedData, actualData);
    }
}
