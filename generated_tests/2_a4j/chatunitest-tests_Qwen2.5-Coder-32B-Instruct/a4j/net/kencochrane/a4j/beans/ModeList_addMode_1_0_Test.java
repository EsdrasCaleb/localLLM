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

public class ModeList_addMode_1_0_Test {

    private ModeList modeList;

    @Mock
    private Mode mockMode;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        modeList = new ModeList();
    }

    @Test
    public void testAddMode() throws NoSuchFieldException, IllegalAccessException {
        // Invoke the addMode method
        modeList.addMode(mockMode);
        // Use reflection to access the private 'modes' field
        Field modesField = ModeList.class.getDeclaredField("modes");
        modesField.setAccessible(true);
        ArrayList<Mode> modes = (ArrayList<Mode>) modesField.get(modeList);
        // Verify that the mode was added to the list
        assertEquals(1, modes.size());
        assertTrue(modes.contains(mockMode));
    }
}
