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

public class Query_SearchQueryGenerator_6_1_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testSearchQueryGenerator() {
        String searchType = "type";
        String searchTerm = "test";
        String mode = "mode";
        String type = "result";
        String page = "1";
        String offer = "offer";
        String expectedQuery = "http://serverURL?t=DSB0XDDW1GQ3S&dev-t=DSB0XDDW1GQ3S&type=test&mode=mode&type=result&page=1&offer=offer&f=xml";
        when(jawsUtil.encodeString(searchTerm)).thenReturn("test");
        String actualQuery = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        assertEquals(expectedQuery, actualQuery);
    }
}
