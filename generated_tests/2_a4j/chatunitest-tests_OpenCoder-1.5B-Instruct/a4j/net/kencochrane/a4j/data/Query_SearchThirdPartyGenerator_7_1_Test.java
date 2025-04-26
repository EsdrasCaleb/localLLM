package net.kencochrane.a4j.data;

import java.lang.reflect.Method;
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

public class Query_SearchThirdPartyGenerator_7_1_Test {

    @Test
    public void testSearchThirdPartyGenerator() throws Exception {
        // Arrange
        String sellerId = "seller123";
        String type = "product";
        String page = "1";
        String status = "active";
        String expectedUrl = "http://example.com?t=seller123&dev-t=DSB0XDDW1GQ3S&SellerSearch=seller123&type=product&page=1&offerstatus=active&f=xml";
        // Create a mock object of the Query class
        Query query = Mockito.mock(Query.class);
        // Get the method to be tested
        Method method = Query.class.getDeclaredMethod("SearchThirdPartyGenerator", String.class, String.class, String.class, String.class);
        // Make the method accessible
        method.setAccessible(true);
        // Call the method and store the result
        String actualUrl = (String) method.invoke(query, sellerId, type, page, status);
        // Assert
        assertEquals(expectedUrl, actualUrl);
    }
}
