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

public class Query_SearchThirdPartyGenerator_7_0_Test {

    @Test
    public void testSearchThirdPartyGenerator_validInput() {
        // Arrange
        Query query = new Query();
        query.serverURL = "https://example.com/api";
        query.associatesID = "12345";
        String sellerId = "seller123";
        String type = "product";
        String page = "1";
        String status = "active";
        // Act
        String result = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        // Assert
        String expected = "https://example.com/api?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=seller123&type=product&page=1&offerstatus=active&f=xml";
        assertEquals(expected, result);
    }

    @Test
    public void testSearchThirdPartyGenerator_nullSellerId() {
        // Arrange
        Query query = new Query();
        query.serverURL = "https://example.com/api";
        query.associatesID = "12345";
        String sellerId = null;
        String type = "product";
        String page = "1";
        String status = "active";
        // Act -  Should not throw NullPointerException
        String result = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        // Assert -  Check for a valid result (e.g., containing "SellerSearch=" if null is handled correctly)
        String expected = "https://example.com/api?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=&type=product&page=1&offerstatus=active&f=xml";
        assertEquals(expected, result);
    }

    @Test
    public void testSearchThirdPartyGenerator_emptySellerId() {
        // Arrange
        Query query = new Query();
        query.serverURL = "https://example.com/api";
        query.associatesID = "12345";
        String sellerId = "";
        String type = "product";
        String page = "1";
        String status = "active";
        // Act
        String result = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        // Assert
        String expected = "https://example.com/api?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=&type=product&page=1&offerstatus=active&f=xml";
        assertEquals(expected, result);
    }
}
