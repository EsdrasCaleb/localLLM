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

    @Test
    void testSearchThirdPartyGenerator() throws NoSuchFieldException, IllegalAccessException {
        Query query = new Query();
        // Using reflection to set private fields, as direct access is not possible.
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://api.example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        String sellerId = "seller123";
        String type = "books";
        String page = "1";
        String status = "active";
        String expectedURL = "https://api.example.com?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=seller123&type=books&page=1&offerstatus=active&f=xml";
        String actualURL = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedURL, actualURL);
        // Test with null values
        sellerId = null;
        type = null;
        page = null;
        status = null;
        expectedURL = "https://api.example.com?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=null&type=null&page=null&offerstatus=null&f=xml";
        actualURL = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedURL, actualURL);
        // Test with empty values
        sellerId = "";
        type = "";
        page = "";
        status = "";
        expectedURL = "https://api.example.com?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=&type=&page=&offerstatus=&f=xml";
        actualURL = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedURL, actualURL);
        // Test with different values
        sellerId = "seller456";
        type = "electronics";
        page = "2";
        status = "inactive";
        expectedURL = "https://api.example.com?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=seller456&type=electronics&page=2&offerstatus=inactive&f=xml";
        actualURL = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedURL, actualURL);
    }
}
