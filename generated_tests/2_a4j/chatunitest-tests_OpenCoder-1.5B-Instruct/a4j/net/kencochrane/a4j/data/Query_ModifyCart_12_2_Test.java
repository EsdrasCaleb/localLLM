package net.kencochrane.a4j.data;

import java.util.ArrayList;
import java.util.List;
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

public class Query_ModifyCart_12_2_Test {

    // Mock object of Query class
    private Query query;

    @Test
    public void testModifyCart() {
        // Arrange
        query = new Query();
        query.serverURL = "http://example.com";
        query.associatesID = "12345";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "item";
        query.type = "book";
        query.page = "1";
        query.offer = "10";
        query.searchValues = new ArrayList<String>();
        // Act
        String result = query.ModifyCart("123", "1", "cart123", "hmac123");
        // Assert
        assertEquals("http://example.com?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Item.123=1&CartId=cart123&Hmac=hmac123", result);
    }
}
