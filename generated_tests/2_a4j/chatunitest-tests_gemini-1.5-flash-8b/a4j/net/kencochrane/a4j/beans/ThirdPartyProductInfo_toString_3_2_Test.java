package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ThirdPartyProductInfo_toString_3_2_Test {

    @Test
    void testToString_withNullProducts() {
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        String expected = "productOffers is null ";
        assertEquals(expected, info.toString());
    }
}

// Dummy class for testing
class ThirdPartyProductDetails {

    private String productName;

    private double price;

    public ThirdPartyProductDetails(String productName, double price) {
        this.productName = productName;
        this.price = price;
    }

    public ThirdPartyProductDetails() {
    }

    @Override
    public String toString() {
        return productName + " " + price;
    }
}
