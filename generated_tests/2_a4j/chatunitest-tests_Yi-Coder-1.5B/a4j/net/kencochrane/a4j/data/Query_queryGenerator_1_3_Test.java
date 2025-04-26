package net.kencochrane.a4j.data;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class Query_queryGenerator_1_3_Test {

    // Test class
    @Test
    public void testQueryGenerator() {
        Query query = new Query();
        ArrayList searchValues = new ArrayList();
        searchValues.add("1");
        searchValues.add("2");
        searchValues.add("3");
        searchValues.add("4");
        searchValues.add("5");
        String queryString = query.queryGenerator("searchType", "type", "1", "1", searchValues);
        assertTrue(queryString.length() > 0);
        assertTrue(queryString.contains("1"));
        assertTrue(queryString.contains("2"));
        assertTrue(queryString.contains("3"));
        assertTrue(queryString.contains("4"));
        assertTrue(queryString.contains("5"));
        assertTrue(queryString.contains("searchType="));
        assertTrue(queryString.contains("type="));
        assertTrue(queryString.contains("offerpage="));
        assertTrue(queryString.contains("offer="));
        assertTrue(queryString.contains("f=xml"));
    }
}
