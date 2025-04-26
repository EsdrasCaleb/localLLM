package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_addAccessory_6_0_Test {

    private FullProduct fullProduct;

    @Mock
    private MiniProduct mockMiniProduct;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        fullProduct = new FullProduct();
    }

    @Test
    void testAddAccessory() throws Exception {
        // Arrange
        Field accessoriesField = FullProduct.class.getDeclaredField("accessories");
        accessoriesField.setAccessible(true);
        ArrayList<MiniProduct> accessories = new ArrayList<>();
        accessoriesField.set(fullProduct, accessories);
        // Act
        fullProduct.addAccessory(mockMiniProduct);
        // Assert
        assertEquals(1, accessories.size());
        assertTrue(accessories.contains(mockMiniProduct));
    }
}
