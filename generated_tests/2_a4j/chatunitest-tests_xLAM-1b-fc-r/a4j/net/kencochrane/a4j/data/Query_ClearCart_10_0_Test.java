package net.kencochrane.a4j.data;

import net.kencochrane.a4j.data.Query;
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

class Query_ClearCart_10_0_Test {

    @Test
    void clearCartTest() {
        Query query = new Query();
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "123456789";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "xml";
        query.type = "clear";
        query.page = "1";
        query.offer = "1";
        query.searchValues = new ArrayList<>();
        String cartId = "1234567890";
        String hmac = "hmacValue";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=123456789&CartId=1234567890&Hmac=hmacValue";
        assertEquals(expectedUrl, query.ClearCart(cartId, hmac));
    }
}
