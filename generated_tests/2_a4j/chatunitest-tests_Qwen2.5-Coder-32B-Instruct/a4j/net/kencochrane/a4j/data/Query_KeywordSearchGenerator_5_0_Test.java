package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
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

public class Query_KeywordSearchGenerator_5_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize protected fields using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    public void testKeywordSearchGenerator() {
        // Arrange
        String searchTerm = "test search";
        String productLine = "books";
        String type = "simple";
        String page = "1";
        String encodedSearchTerm = "test+search";
        when(jawsUtil.encodeString(searchTerm)).thenReturn(encodedSearchTerm);
        // Act
        String result = query.KeywordSearchGenerator(searchTerm, productLine, type, page);
        // Assert
        String expectedUrl = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test+search&mode=books&type=simple&page=1&f=xml";
        assertEquals(expectedUrl, result);
        // Verify
        verify(jawsUtil, times(1)).encodeString(searchTerm);
    }
}
