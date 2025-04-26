package com.hf.sfm.sfmis.personinfo.business;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.pdo.APersonInfo;
import com.hf.sfm.util.DaoFactory;

@ExtendWith(MockitoExtension.class)
public class PersonInfoMgr_deleteByIds_1_1_Test {

    @Mock
    private PersonInfoMgr focalClass;

    @Test
    public void testDeleteByIds() {
        String[] idnos = { "1", "2", "3" };
        String expectedResult = "1";
        when(focalClass.deleteByIds(idnos)).thenReturn(expectedResult);
        String actualResult = focalClass.deleteByIds(idnos);
        assertEquals(expectedResult, actualResult);
        verify(focalClass, times(1)).deleteByIds(idnos);
    }

    @Test
    public void testDeleteByIdsFailure() {
        String[] idnos = { "1", "2", "3" };
        String expectedResult = "0";
        when(focalClass.deleteByIds(idnos)).thenReturn(expectedResult);
        String actualResult = focalClass.deleteByIds(idnos);
        assertEquals(expectedResult, actualResult);
        verify(focalClass, times(1)).deleteByIds(idnos);
    }
}
