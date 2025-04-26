// Test method
package com.hf.sfm.system.business;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.hf.sfm.sfmis.personinfo.business.PersonInfoMgr;
import com.hf.sfm.system.pdo.AWorker;
import com.hf.sfm.util.DaoFactory;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.hibernate.Transaction;

@ExtendWith(MockitoExtension.class)
public class WorkerMgr_deleteByIds_1_1_Test {

    @Mock
    private WorkerMgr workerMgr;

    @Test
    public void testDeleteByIds() {
        String[] ids = { "1", "2", "3" };
        when(workerMgr.deleteByIds(ids)).thenReturn("1");
        Assertions.assertEquals("1", workerMgr.deleteByIds(ids));
        verify(workerMgr, times(1)).deleteByIds(ids);
    }
}
