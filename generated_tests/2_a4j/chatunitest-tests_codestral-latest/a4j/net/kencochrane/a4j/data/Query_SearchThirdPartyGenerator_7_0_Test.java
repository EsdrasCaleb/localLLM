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

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Set up the protected fields using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        Field searchTypeField = Query.class.getDeclaredField("searchType");
        searchTypeField.setAccessible(true);
        searchTypeField.set(query, "type");
        Field typeField = Query.class.getDeclaredField("type");
        typeField.setAccessible(true);
        typeField.set(query, "type");
        Field pageField = Query.class.getDeclaredField("page");
        pageField.setAccessible(true);
        pageField.set(query, "1");
        Field offerField = Query.class.getDeclaredField("offer");
        offerField.setAccessible(true);
        offerField.set(query, "offer");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    public void testSearchThirdPartyGenerator() {
        String sellerId = "seller123";
        String type = "type";
        String page = "1";
        String status = "active";
        String expectedURL = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=seller123&type=type&page=1&offerstatus=active&f=xml";
        String actualURL = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedURL, actualURL);
    }
}
