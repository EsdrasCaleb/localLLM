package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_printFullProduct_8_0_Test {

    @Test
    public void testPrintFullProduct_NoAccessories_NoSimilarItems() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(new ProductDetails());
        fullProduct.setAccessories(null);
        fullProduct.setSimilarItems(null);
        fullProduct.printFullProduct();
    }

    @Test
    public void testPrintFullProduct_NoAccessories_HasSimilarItems() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(new ProductDetails());
        fullProduct.setAccessories(null);
        fullProduct.setSimilarItems(new ArrayList<>());
        fullProduct.getSimilarItems().add("Similar Item 1");
        fullProduct.printFullProduct();
    }

    @Test
    public void testPrintFullProduct_HasAccessories_NoSimilarItems() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(new ProductDetails());
        fullProduct.setAccessories(new ArrayList<>());
        fullProduct.setSimilarItems(null);
        fullProduct.printFullProduct();
    }

    @Test
    public void testPrintFullProduct_HasAccessories_HasSimilarItems() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(new ProductDetails());
        fullProduct.setAccessories(new ArrayList<>());
        fullProduct.setSimilarItems(new ArrayList<>());
        fullProduct.getSimilarItems().add("Similar Item 1");
        fullProduct.getSimilarItems().add("Similar Item 2");
        fullProduct.printFullProduct();
    }

    @Test
    public void testPrintFullProduct_NullDetails() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setAccessories(new ArrayList<>());
        fullProduct.setSimilarItems(new ArrayList<>());
        assertThrows(NullPointerException.class, () -> fullProduct.printFullProduct());
    }

    @Test
    public void testPrintFullProduct_NullAccessories() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(new ProductDetails());
        fullProduct.setSimilarItems(new ArrayList<>());
        assertThrows(NullPointerException.class, () -> fullProduct.printFullProduct());
    }

    @Test
    public void testPrintFullProduct_NullSimilarItems() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(new ProductDetails());
        fullProduct.setAccessories(new ArrayList<>());
        assertThrows(NullPointerException.class, () -> fullProduct.printFullProduct());
    }
}
