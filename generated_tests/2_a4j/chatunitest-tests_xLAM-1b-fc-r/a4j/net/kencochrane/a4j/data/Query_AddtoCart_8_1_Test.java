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

public class Query_AddtoCart_8_1_Test {

    @Test
    public void testAddtoCart() {
        Query query = new Query();
        query.serverURL = "http://example.com";
        query.associatesID = "123456789";
        query.type = "product";
        query.offer = "1234567890";
        query.searchType = "ASIN";
        query.searchValues = new ArrayList<>();
        query.searchValues.add("ASIN");
        query.searchValues.add("123456789");
        query.searchValues.add("1234567890");
        StringBuffer buffer = new StringBuffer();
        buffer.append(query.serverURL);
        buffer.append("?");
        buffer.append("ShoppingCart=add&f=xml&dev-t=");
        buffer.append(query.token);
        buffer.append("&t=");
        buffer.append(query.associatesID);
        buffer.append("&Asin.");
        buffer.append(query.searchValues.get(1));
        buffer.append("=");
        buffer.append(query.searchValues.get(2));
        Assertions.assertEquals(buffer.toString(), query.AddtoCart("123456789", "1"));
    }
}
