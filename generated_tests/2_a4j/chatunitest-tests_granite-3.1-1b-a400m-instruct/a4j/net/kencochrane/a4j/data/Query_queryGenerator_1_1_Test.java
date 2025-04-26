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

public class Query_queryGenerator_1_1_Test {

    @Test
    public void testQueryGenerator() {
        // Arrange
        String serverURL = "http://example.com/search";
        String associatesID = "12345";
        String token = "A4J-0123456789";
        String searchType = "all";
        String type = "products";
        String page = "1";
        String offer = "premium";
        ArrayList searchValues = new ArrayList<>();
        searchValues.add("product1");
        searchValues.add("product2");
        Query query = new Query();
        // Act
        String result = query.queryGenerator(searchType, type, page, offer, searchValues);
        // Assert
        assertEquals("http://example.com/search?t=12345&dev-t=A4J-0123456789&searchType=all&type=products&page=1&offerpage=1&offer=premium&f=xml", result);
    }
}
