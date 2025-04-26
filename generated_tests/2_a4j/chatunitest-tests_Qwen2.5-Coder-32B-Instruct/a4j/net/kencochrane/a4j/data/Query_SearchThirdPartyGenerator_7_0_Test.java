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

public class Query_SearchThirdPartyGenerator_7_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        query = new Query();
        setField(query, "serverURL", "http://example.com/search");
        setField(query, "associatesID", "12345");
        setField(query, "searchValues", new ArrayList<>());
        setField(query, "jawsUtil", jawsUtil);
    }

    private void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    @Test
    public void testSearchThirdPartyGenerator() {
        String sellerId = "seller123";
        String type = "books";
        String page = "1";
        String status = "active";
        String expectedUrl = "http://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=seller123&type=books&page=1&offerstatus=active&f=xml";
        String actualUrl = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedUrl, actualUrl);
    }
}
