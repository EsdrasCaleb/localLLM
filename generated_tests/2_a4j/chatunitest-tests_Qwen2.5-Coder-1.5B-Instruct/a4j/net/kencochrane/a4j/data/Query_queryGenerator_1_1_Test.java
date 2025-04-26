package net.kencochrane.a4j.data;

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

class Query_queryGenerator_1_1_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testQueryGenerator() {
        // Arrange
        String searchType = "search";
        String type = "type";
        String page = "page";
        String offer = "offer";
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add("value1");
        searchValues.add("value2");
        // Act
        String result = query.queryGenerator(searchType, type, page, offer, searchValues);
        // Assert
        assertEquals("http://example.com?t=your_associates_id&dev-t=DSB0XDDW1GQ3S&search=search=value1,value2&type=type&page=page&offer=offer&f=xml", result);
    }
}
