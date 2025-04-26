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

    @BeforeEach
    public void setUp() {
        modeList = new ModeList();
    }

    @Test
    public void testAddMode() throws Exception {
        // Arrange
        Mode mode = mock(Mode.class);
        // Act
        modeList.addMode(mode);
        // Assert
        ArrayList modes = getModesField(modeList);
        assertEquals(1, modes.size());
        assertEquals(mode, modes.get(0));
    }

    @SuppressWarnings("unchecked")
    private ArrayList<Mode> getModesField(ModeList modeList) throws Exception {
        Field field = ModeList.class.getDeclaredField("modes");
        field.setAccessible(true);
        return (ArrayList<Mode>) field.get(modeList);
    }
}
