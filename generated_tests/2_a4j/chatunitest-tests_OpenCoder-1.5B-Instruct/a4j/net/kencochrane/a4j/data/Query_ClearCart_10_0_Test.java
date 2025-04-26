package net.kencochrane.a4j.data;

import java.util.HashMap;
import java.util.Map;
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

public class Query_ClearCart_10_0_Test {

    @Test
    public void testClearCart() {
        Query query = new Query();
        query.serverURL = "http://example.com";
        query.associatesID = "12345";
        query.ClearCart("cart1", "hmac123");
        assertEquals("http://example.com?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&CartId=cart1&Hmac=hmac123", query.ClearCart("cart1", "hmac123"));
    }
}
