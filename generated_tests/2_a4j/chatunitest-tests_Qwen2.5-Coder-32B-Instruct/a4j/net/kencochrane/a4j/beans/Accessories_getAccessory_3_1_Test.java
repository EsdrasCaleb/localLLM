package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Accessories_getAccessory_3_1_Test {

    @InjectMocks
    private Accessories accessories;

    @Mock
    private ArrayList<String> mockAccessoryList;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the actual ArrayList instead of the mock for this test
        Field field = Accessories.class.getDeclaredField("accessory");
        field.setAccessible(true);
        field.set(accessories, new ArrayList<>());
    }

    @Test
    public void testGetAccessory_WhenIndexIsNegative_ReturnsNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        ArrayList<String> testList = new ArrayList<>();
        testList.add("Hat");
        testList.add("Gloves");
        testList.add("Boots");
        Field field = Accessories.class.getDeclaredField("accessory");
        field.setAccessible(true);
        field.set(accessories, testList);
        // Act
        String result = accessories.getAccessory(-1);
        // Assert
        assertNull(result);
    }
}
