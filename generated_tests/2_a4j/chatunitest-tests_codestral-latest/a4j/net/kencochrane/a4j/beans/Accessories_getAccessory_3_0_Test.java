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

class Accessories_getAccessory_3_0_Test {

    @InjectMocks
    private Accessories accessories;

    @Mock
    private ArrayList<String> mockAccessory;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        accessories.setAccessory(new String[] { "Watch", "Bag", "Shoes" });
    }

    @Test
    void testGetAccessoryNegativeIndex() {
        String result = accessories.getAccessory(-1);
        assertNull(result);
    }
}
