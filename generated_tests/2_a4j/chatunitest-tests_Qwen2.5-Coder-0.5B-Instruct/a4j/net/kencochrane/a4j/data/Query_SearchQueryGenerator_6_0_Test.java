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

class Query_SearchQueryGenerator_6_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @Test
    public void testSearchQueryGenerator() {
        // Arrange
        Query query = new Query();
        String searchType = "example";
        String searchTerm = "test";
        String mode = "test";
        String type = "test";
        String page = "test";
        String offer = "test";
        // Act
        String result = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        // Assert
        assertEquals("https://example.com/?t=example&associatesID=test&dev-t=test&searchType=example&mode=test&type=test&page=test&offer=test&f=xml", result);
    }
}
