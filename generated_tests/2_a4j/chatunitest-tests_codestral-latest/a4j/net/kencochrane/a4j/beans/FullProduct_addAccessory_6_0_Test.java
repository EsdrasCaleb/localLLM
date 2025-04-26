package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class FullProduct_addAccessory_6_0_Test {

    @InjectMocks
    private FullProduct fullProduct;

    @Mock
    private MiniProduct miniProduct;

    @BeforeEach
    public void setUp() {
        fullProduct.setAccessories(new ArrayList<>());
    }

    @Test
    public void testAddAccessory() {
        fullProduct.addAccessory(miniProduct);
        ArrayList<MiniProduct> accessories = fullProduct.getAccessories();
        assertEquals(1, accessories.size());
        assertTrue(accessories.contains(miniProduct));
    }

    @Test
    public void testAddAccessoryMultiple() {
        MiniProduct miniProduct2 = mock(MiniProduct.class);
        fullProduct.addAccessory(miniProduct);
        fullProduct.addAccessory(miniProduct2);
        ArrayList<MiniProduct> accessories = fullProduct.getAccessories();
        assertEquals(2, accessories.size());
        assertTrue(accessories.contains(miniProduct));
        assertTrue(accessories.contains(miniProduct2));
    }
}
