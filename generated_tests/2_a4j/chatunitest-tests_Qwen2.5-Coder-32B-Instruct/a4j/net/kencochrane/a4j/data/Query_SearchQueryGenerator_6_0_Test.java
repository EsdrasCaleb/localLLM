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

public class Query_SearchQueryGenerator_6_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        setFieldValue(query, "serverURL", "http://example.com");
        setFieldValue(query, "associatesID", "12345");
    }

    private void setFieldValue(Object obj, String fieldName, Object value) throws Exception {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }

    @Test
    public void testSearchQueryGenerator() throws Exception {
        // Arrange
        String searchType = "title";
        String searchTerm = "Java Programming";
        String mode = "advanced";
        String type = "books";
        String page = "1";
        String offer = "discount";
        String encodedSearchTerm = "Java%20Programming";
        when(jawsUtil.encodeString(searchTerm)).thenReturn(encodedSearchTerm);
        // Expected URL
        String expectedUrl = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&title=Java%20Programming&mode=advanced&type=books&page=1&offer=discount&f=xml";
        // Act
        String result = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        // Assert
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(searchTerm);
    }
}
