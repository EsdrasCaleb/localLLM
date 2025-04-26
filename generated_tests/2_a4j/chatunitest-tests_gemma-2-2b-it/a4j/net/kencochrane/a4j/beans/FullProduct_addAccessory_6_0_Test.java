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

public class FullProduct_addAccessory_6_0_Test {

    @Test
    void addAccessory() {
        FullProduct fullProduct = Mockito.mock(FullProduct.class);
        MiniProduct miniProduct = Mockito.mock(MiniProduct.class);
        when(fullProduct.getAccessories()).thenReturn(new ArrayList<>());
        fullProduct.addAccessory(miniProduct);
        List<MiniProduct> accessories = fullProduct.getAccessories();
        assertEquals(1, accessories.size());
        assertTrue(accessories.contains(miniProduct));
    }
}
