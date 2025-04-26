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

public class Query_SearchQueryGenerator_6_2_Test {

    @Test
    public void testSearchQueryGenerator() {
        // Setup
        String searchType = "news";
        String searchTerm = "Apple";
        String mode = "full";
        String type = "article";
        String page = "1";
        String offer = "premium";
        // Arrange
        Query query = new Query();
        String expectedQuery = "serverURL?t=associatesID&dev-t=DSB0XDDW1GQ3S&mode=full&type=article&page=1&offer=premium";
        // Act
        String actualQuery = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        // Assert
        assertEquals(expectedQuery, actualQuery);
    }
}
