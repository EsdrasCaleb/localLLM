package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class ModeList_addMode_1_0_Test {

    @InjectMocks
    private ModeList modeList;

    @Test
    public void testAddMode() {
        // Arrange
        Mode mode = new Mode();
        modeList.addMode(mode);
        // Act
        ArrayList modes = modeList.getAllModes();
        // Assert
        assertEquals(1, modes.size());
    }

    @Test
    public void testAddModeTwice() {
        // Arrange
        Mode mode1 = new Mode();
        Mode mode2 = new Mode();
        modeList.addMode(mode1);
        modeList.addMode(mode2);
        // Act
        ArrayList modes = modeList.getAllModes();
        // Assert
        assertEquals(2, modes.size());
    }

    @Test
    public void testAddModeNull() {
        // Arrange
        Mode mode = null;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> modeList.addMode(mode));
    }
}
