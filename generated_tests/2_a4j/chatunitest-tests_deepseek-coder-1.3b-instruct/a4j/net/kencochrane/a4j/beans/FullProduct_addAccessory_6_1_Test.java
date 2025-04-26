package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class FullProduct_addAccessory_6_1_Test {

    FullProduct fullProduct = new FullProduct();

    @Test
    public void testAddAccessory() {
        MiniProduct product = new MiniProduct();
        fullProduct.addAccessory(product);
        assertEquals(1, fullProduct.getAccessories().size());
    }
}
