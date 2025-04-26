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

    @Test
    void testToString_withEmptyProducts() {
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        ThirdPartyProductDetails[] emptyArray = new ThirdPartyProductDetails[0];
        info.setThirdPartyProductDetails(emptyArray);
        String expected = "productOffers is null ";
        assertEquals(expected, info.toString());
    }

    @Test
    void testToString_withProducts() {
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        ThirdPartyProductDetails product1 = new ThirdPartyProductDetails("Product 1", 10.0);
        ThirdPartyProductDetails product2 = new ThirdPartyProductDetails("Product 2", 20.0);
        ThirdPartyProductDetails[] products = { product1, product2 };
        info.setThirdPartyProductDetails(products);
        String expected = product1 + "\n" + product2 + "\n# of productOffers = 2";
        assertEquals(expected, info.toString());
    }

    // Test with a more comprehensive product list.
    @Test
    void testToString_withMultipleProducts() {
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        ThirdPartyProductDetails[] products = new ThirdPartyProductDetails[5];
        for (int i = 0; i < 5; i++) {
            products[i] = new ThirdPartyProductDetails("Product " + (i + 1), (double) (i + 1) * 10);
        }
        info.setThirdPartyProductDetails(products);
        String expected = "";
        for (ThirdPartyProductDetails product : products) {
            expected += product + "\n";
        }
        expected += "# of productOffers = 5";
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
