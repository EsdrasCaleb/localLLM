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

public class Query_RemoveFromCart_13_0_Test {

    // Focal method
    public String RemoveFromCart(String itemId, String cartId, String hmac) {
        StringBuffer buffer = new StringBuffer();
        buffer.append("http://xml.amazon.com/onca/xml3?");
        buffer.append("ShoppingCart=remove");
        buffer.append("&CartId=");
        buffer.append(cartId);
        buffer.append("&Hmac=");
        buffer.append(hmac);
        buffer.append("&Item.");
        buffer.append(itemId);
        buffer.append("&dev-t=");
        buffer.append("DSB0XDDW1GQ3S");
        buffer.append("&t=");
        buffer.append("test");
        buffer.append("&f=xml");
        buffer.append("&type=lite");
        return new String(buffer);
    }

    @Test
    public void testRemoveFromCart() {
        Query query = mock(Query.class);
        String itemId = "17120277375791359165";
        String cartId = "1813019710362345961";
        String hmac = "HMAC";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&CartId=1813019710362345961&Hmac=HMAC&Item.17120277375791359165&dev-t=DSB0XDDW1GQ3S&t=test&f=xml&type=lite";
        String actualUrl = query.RemoveFromCart(itemId, cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
    }
}
