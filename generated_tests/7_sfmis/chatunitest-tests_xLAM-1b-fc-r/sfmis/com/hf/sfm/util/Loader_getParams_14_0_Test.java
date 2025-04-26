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

class Loader_getParams_14_0_Test {

    @Test
    void testGetParams() {
        // Arrange
        String querySql = "SELECT * FROM table WHERE column = ?";
        String queryValue = "value";
        String[][] params = { { queryValue }, { "String" } };
        Loader loader = new Loader();
        // Act
        Query result = loader.getParams(Mockito.mock(Query.class), params);
        // Assert
        Mockito.verify(result, Mockito.times(1)).setParameter(Mockito.eq(0), Mockito.eq(queryValue));
    }
}
