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

public class Accessories_getAccessory_3_0_Test {

    private Accessories accessories;

    @BeforeEach
    void setUp() {
        accessories = new Accessories();
    }

    @Test
    void testGetAccessoryNegativeIndex() {
        String[] accessoriesArray = { "Hat", "Glove", "Shoes" };
        accessories.setAccessory(accessoriesArray);
        assertNull(accessories.getAccessory(-1));
    }
}
