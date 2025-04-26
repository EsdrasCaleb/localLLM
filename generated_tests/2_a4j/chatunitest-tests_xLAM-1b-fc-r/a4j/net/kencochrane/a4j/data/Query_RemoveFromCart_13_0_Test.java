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

public class Query_RemoveFromCart_13_0_Test {

    @Test
    public void testRemoveFromCart() {
        Query query = new Query();
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "test";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "type";
        query.type = "lite";
        query.page = "1";
        query.offer = "1";
        query.searchValues = new ArrayList<>();
        query.searchValues.add("item1");
        query.searchValues.add("item2");
        String itemId = "17120277375791359165";
        String cartId = "CART";
        String hmac = "HMAC";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&dev-t=DSB0XDDW1GQ3S&t=test&f=xml&type=lite&Item." + itemId + "&CartId=CART&Hmac=HMAC";
        assertEquals(expectedUrl, query.RemoveFromCart(itemId, cartId, hmac));
    }
}
