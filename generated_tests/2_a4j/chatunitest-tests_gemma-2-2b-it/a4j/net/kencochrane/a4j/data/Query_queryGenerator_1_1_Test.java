package net.kencochrane.a4j.data;

import static org.mockito.ArgumentMatchers.*;
import java.util.ArrayList;
import java.util.List;
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
import java.util.Properties;

public class Query_queryGenerator_1_1_Test {

    @Test
    void testQueryGenerator() {
        // Arrange
        Query query = new Query();
        String searchType = "test";
        String type = "type";
        String page = "page";
        String offer = "offer";
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add("value1");
        searchValues.add("value2");
        // Act
        String generatedQuery = query.queryGenerator(searchType, type, page, offer, searchValues);
        // Assert
        assertEquals(generatedQuery, "http://localhost:8080/api/search?t=associatesID&dev-t=DSB0XDDW1GQ3S&searchType=test=value1&type=type&offerpage=page&offer=offer&f=xml");
    }
}
