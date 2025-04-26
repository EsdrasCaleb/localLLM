package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class FullProduct_printFullProduct_8_1_Test {

    @Test
    void printFullProduct() {
        FullProduct fullProduct = mock(FullProduct.class);
        when(fullProduct.getDetails()).thenReturn(new ProductDetails());
        when(fullProduct.getAccessories()).thenReturn(new ArrayList<>());
        when(fullProduct.getSimilarItems()).thenReturn(new ArrayList<>());
        fullProduct.printFullProduct();
    }
}
