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

class Query_SearchThirdPartyGenerator_7_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.openMocks(this);
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com/query");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "ABC123");
        Field tokenField = Query.class.getDeclaredField("token");
        tokenField.setAccessible(true);
        tokenField.set(query, "DSB0XDDW1GQ3S");
        Field searchTypeField = Query.class.getDeclaredField("searchType");
        searchTypeField.setAccessible(true);
        searchTypeField.set(query, "products");
        Field typeField = Query.class.getDeclaredField("type");
        typeField.setAccessible(true);
        typeField.set(query, "active");
        Field pageField = Query.class.getDeclaredField("page");
        pageField.setAccessible(true);
        pageField.set(query, "1");
        Field offerField = Query.class.getDeclaredField("offer");
        offerField.setAccessible(true);
        offerField.set(query, "pending");
    }

    @Test
    public void testSearchThirdPartyGenerator() {
        // Arrange
        String sellerId = "XYZ789";
        String type = "inactive";
        String page = "2";
        String status = "completed";
        // Act
        String actualUrl = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        // Assert
        assertEquals("https://example.com/query?t=ABC123&dev-t=DSB0XDDW1GQ3S&SellerSearch=XYZ789&type=inactive&page=2&offerstatus=completed&f=xml", actualUrl);
    }
}
