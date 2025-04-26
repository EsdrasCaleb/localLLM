package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class FullProduct_addAccessory_6_0_Test {

    @Test
    void testAddAccessory() {
        FullProduct fullProduct = new FullProduct();
        fullProduct.addAccessory(new MiniProduct());
        assertTrue(fullProduct.getAccessories().contains(new MiniProduct()));
    }
}
