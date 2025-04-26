package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductInfo_toString_9_3_Test {

    @Test
    void testToString_withoutProducts() throws NoSuchFieldException, IllegalAccessException {
        ProductInfo productInfo = new ProductInfo();
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, null);
        productInfo.setTotalResults("0");
        productInfo.setTotalPages("0");
        String expectedOutput = "Total results = 0\n" + "Total pages = 0\n" + "products is null \n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    void testToString_emptyProducts() throws NoSuchFieldException, IllegalAccessException {
        ProductInfo productInfo = new ProductInfo();
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, new ArrayList<>());
        productInfo.setTotalResults("0");
        productInfo.setTotalPages("0");
        String expectedOutput = "Total results = 0\n" + "Total pages = 0\n" + "# of products = 0\n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    class ProductDetails {

        private String name;

        private String price;

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getPrice() {
            return price;
        }

        public void setPrice(String price) {
            this.price = price;
        }

        @Override
        public String toString() {
            return "Name: " + name + ", Price: " + price;
        }
    }
}
